# Harmonisation Plan

## Status

No final harmonisation protocol is recommended until acquisition metadata has
been verified for every cohort.

## Evidence-based starting configuration

- Keep raw recordings and participant-level provenance unchanged.
- Use 19 standard 10-20 scalp locations as the provisional common channel
  configuration.
- Normalize legacy temporal names T3, T4, T5, and T6 to T7, T8, P7, and P8.
- Do not infer ADFSU duration or sampling frequency from sample counts.
- Run input-source and montage ablations only after acquisition metadata is
  sufficient to define each candidate independently.

## Alternatives requiring separate evidence

1. Common-channel subset based on observed availability.
2. Montage-based comparison using verified coordinates.
3. Higher-density source-specific pipelines evaluated without cross-cohort tuning.

This plan deliberately does not prescribe a final resampling frequency,
reference, filter, epoch duration, or exclusion threshold.
