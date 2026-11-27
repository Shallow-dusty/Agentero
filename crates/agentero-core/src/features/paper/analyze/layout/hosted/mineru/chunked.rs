//! Chunked cloud layout only; whole-document and self-hosted callers keep their transports.

use super::{
    parse_cloud_layout, parse_layout_json, read_zip_entry_by_candidates, resolve_target,
    run_cloud_extract, AppError, HostedProviderCredentials, LayoutRemoteAnalyzePdfResult,
    MIDDLE_CANDIDATES, MINERU_REQUEST_TIMEOUT,
};
use crate::http;
use bytes::Bytes;
use serde_json::Value;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

pub(super) const PAGES_PER_SEGMENT: usize = 200;
// Each job uploads the whole PDF again; cap analysis at ten segments.
const MAX_PAGES: usize = 2_000;
// Cumulative decompressed layout JSON, in addition to the existing per-file/ZIP caps.
const MAX_MERGED_JSON_BYTES: usize = 256 * 1024 * 1024;
const CANCELLED: &str = "layout analysis cancelled";

#[cfg(not(any(target_os = "ios", target_os = "android")))]
pub(super) fn pdf_page_count(pdf: &[u8]) -> Option<u32> {
    let library = crate::features::pdf::open_library().ok()?;
    let document = library.load_document_from_bytes(pdf, None).ok()?;
    u32::try_from(document.page_count()).ok().filter(|n| *n > 0)
}

pub(super) async fn analyze(
    credentials: &HostedProviderCredentials,
    pdf: Bytes,
    file_name: &str,
    page_count: usize,
    progress: super::super::ProgressFn<'_>,
    cancel: super::super::CancelFn<'_>,
) -> Result<LayoutRemoteAnalyzePdfResult, AppError> {
    if page_count > MAX_PAGES {
        return Err(AppError::message(format!(
            "MinerU cloud layout in Agentero supports up to {MAX_PAGES} pages; PDF has {page_count}"
        )));
    }
    let (base, auth) = resolve_target(credentials)?;
    let client = http::client(MINERU_REQUEST_TIMEOUT)?;
    let mut result = LayoutRemoteAnalyzePdfResult {
        pages: Vec::with_capacity(page_count),
        rendered_pages: Vec::with_capacity(page_count),
    };
    let mut remaining_json_bytes = MAX_MERGED_JSON_BYTES;
    let segments = page_count.div_ceil(PAGES_PER_SEGMENT);
    let total = Some(page_count as u64);
    for (segment, start) in (0..page_count).step_by(PAGES_PER_SEGMENT).enumerate() {
        if cancel() {
            return Err(AppError::message(CANCELLED));
        }
        let end = (start + PAGES_PER_SEGMENT).min(page_count);
        let page_ranges = format!("{}-{end}", start + 1);
        let segment_pages = (end - start) as u64;
        let extracted_pages = AtomicU64::new(0);
        let report = |_: &str, extracted: Option<u64>, reported_total: Option<u64>| {
            // A job may report the original document total even though it
            // processes only this range. Cap its contribution at this segment.
            if let Some(n) = extracted
                .filter(|_| reported_total == Some(segment_pages) || reported_total == total)
            {
                extracted_pages.fetch_max(n.min(segment_pages), Ordering::Relaxed);
            }
            // Per-job uploading/downloading presets in the viewer would reset
            // the bar. Report cumulative document progress through its running state.
            progress(
                "running",
                Some(start as u64 + extracted_pages.load(Ordering::Relaxed)),
                total,
            );
        };
        let zip = run_cloud_extract(
            &client,
            &base,
            &auth,
            pdf.clone(),
            file_name,
            Some(&page_ranges),
            credentials,
            &report,
            cancel,
        )
        .await;
        if cancel() {
            return Err(AppError::message(CANCELLED));
        }
        let part = zip
            .and_then(|zip| parse_segment(&zip, start..end, &mut remaining_json_bytes))
            .map_err(|error| {
                AppError::message(format!(
                    "MinerU segment {}/{segments} (pages {page_ranges}) failed: {error}",
                    segment + 1
                ))
            })?;
        result.pages.extend(part.pages);
        result.rendered_pages.extend(part.rendered_pages);
        progress("running", Some(end as u64), total);
    }
    if cancel() {
        return Err(AppError::message(CANCELLED));
    }
    Ok(result)
}

fn parse_segment(
    zip: &[u8],
    range: Range<usize>,
    remaining_json_bytes: &mut usize,
) -> Result<LayoutRemoteAnalyzePdfResult, AppError> {
    let content = read_zip_entry_by_candidates(zip, &["content_list.json"])?;
    let middle = read_zip_entry_by_candidates(zip, MIDDLE_CANDIDATES)?;
    *remaining_json_bytes = remaining_json_bytes
        .checked_sub(content.len() + middle.len())
        .ok_or_else(|| AppError::message("MinerU combined layout JSON too large"))?;
    let mut content = parse_layout_json(&content, "content_list")?;
    let middle = parse_layout_json(&middle, "middle")?;
    let pages = middle
        .get("pdf_info")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::message("MinerU result is missing pdf_info"))?;
    if pages.len() != range.len() {
        return Err(AppError::message(format!(
            "MinerU returned {} page records, expected {}",
            pages.len(),
            range.len()
        )));
    }
    // Missing indices keep their array position, as in the whole-document
    // parser. Every explicit index must agree with that position and one base.
    let mut bases = [0, range.start]
        .into_iter()
        .filter(|&base| {
            pages.iter().enumerate().all(|(i, page)| {
                page.is_object()
                    && match page.get("page_idx") {
                        None | Some(Value::Null) => true,
                        Some(index) => index.as_u64() == Some((base + i) as u64),
                    }
            })
        })
        .peekable();
    if bases.peek().is_none() {
        return Err(AppError::message(
            "MinerU result has unexpected pdf_info page indices",
        ));
    }
    let blocks = content
        .as_array_mut()
        .ok_or_else(|| AppError::message("MinerU content_list is not an array"))?;
    let local_index = |block: &Value, base: usize| {
        block
            .get("page_idx")
            .and_then(Value::as_u64)
            .and_then(|index| index.checked_sub(base as u64))
            .filter(|&index| index < range.len() as u64)
    };
    // After the first segment the relative and absolute ranges are disjoint.
    // Content List can resolve the base when pdf_info has no explicit indices.
    let index_base = bases
        .find(|&base| {
            blocks
                .iter()
                .all(|block| local_index(block, base).is_some())
        })
        .ok_or_else(|| {
            AppError::message("MinerU content_list page index does not match pdf_info")
        })?;
    if index_base != 0 {
        for block in blocks {
            block["page_idx"] = local_index(block, index_base)
                .expect("page index was validated")
                .into();
        }
    }
    // Both output arrays are dense and in range order. Appending once applies
    // the document offset to boxes, captions, blank pages and sizes together.
    parse_cloud_layout(&content, &middle)
}
