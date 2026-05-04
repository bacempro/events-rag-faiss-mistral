# Clean OpenAgenda Events Dataset Schema

## Purpose

This file defines the target schema for the cleaned OpenAgenda event dataset used by the Puls-Events RAG POC.

Raw input:

```text
data/raw/openagenda_events_raw.json
```

Clean output:

```text
data/processed/events_clean.csv
```

The raw extraction already applies the main API-side filters:

- City: Paris
- Event timing lower bound: today - 365 days
- Future events included

The preprocessing step must not be responsible for the primary one-year filtering rule. It may only perform defensive validation and remove malformed records if required.

---

## Target columns

| Column | Type | Required | Source field | Purpose |
|---|---:|---:|---|---|
| `event_id` | integer | yes | `uid` | Stable OpenAgenda event identifier. |
| `slug` | string | no | `slug` | Human-readable event slug. |
| `title` | string | yes | `title.fr` | Event title in French. |
| `description` | string | yes | `description.fr` | Short event description in French. |
| `long_description` | string | no | `longDescription.fr` | Long event description in French. |
| `date_range` | string | no | `dateRange.fr` | Human-readable date range. |
| `first_timing_begin` | datetime string | yes | `firstTiming.begin` | First known event start datetime. |
| `first_timing_end` | datetime string | no | `firstTiming.end` | First known event end datetime. |
| `last_timing_begin` | datetime string | no | `lastTiming.begin` | Last known event start datetime. |
| `last_timing_end` | datetime string | no | `lastTiming.end` | Last known event end datetime. |
| `next_timing_begin` | datetime string | no | `nextTiming.begin` | Next upcoming timing start, when available. |
| `next_timing_end` | datetime string | no | `nextTiming.end` | Next upcoming timing end, when available. |
| `timings_count` | integer | yes | `timings` | Number of available timing slots. |
| `city` | string | yes | `location.city` | Event city. Expected: Paris. |
| `postal_code` | string | yes | `location.postalCode` | Event postal code. |
| `district` | string | no | `location.district` | Paris district/arrondissement when available. |
| `department` | string | no | `location.department` | Department name. |
| `region` | string | no | `location.region` | Region name. |
| `country_code` | string | yes | `location.countryCode` | Country code. Expected: FR. |
| `venue_name` | string | yes | `location.name` | Event venue name. |
| `address` | string | yes | `location.address` | Event venue address. |
| `latitude` | float | yes | `location.latitude` | Venue latitude. |
| `longitude` | float | yes | `location.longitude` | Venue longitude. |
| `conditions` | string | no | `conditions.fr` | Booking/access conditions. |
| `registration_email` | string | no | `registration` | Contact email when available. |
| `registration_url` | string | no | `registration` | Registration or booking URL when available. |
| `origin_agenda_uid` | integer | no | `originAgenda.uid` | Source agenda UID. |
| `origin_agenda_title` | string | no | `originAgenda.title` | Source agenda title. |
| `source` | string | yes | constant | Data source marker: `OpenAgenda`. |
| `text_for_embedding` | string | yes | derived | Concatenated clean text for Step 3 vectorization. |

---

## Required columns for downstream RAG

The following columns are required for Step 3 vectorization and Step 4 RAG responses:

```text
event_id
title
description
first_timing_begin
city
venue_name
address
latitude
longitude
text_for_embedding
```

Rows missing any of these fields should be rejected during preprocessing.

---

## `text_for_embedding` construction

The `text_for_embedding` field should be built from available French fields and useful metadata.

Recommended order:

```text
Titre: {title}
Description: {description}
Description détaillée: {long_description}
Dates: {date_range}
Lieu: {venue_name}, {address}, {postal_code} {city}
Conditions: {conditions}
```

Rules:

- Include only non-empty fields.
- Normalize whitespace.
- Avoid raw JSON fragments.
- Keep French content as the primary language.
- Do not translate content.
- Do not call Mistral or any embedding model in Step 2.

---

## Preprocessing rules

### Keep

Keep events that have:

- a valid `uid`
- a non-empty `title.fr`
- a non-empty `description.fr`
- at least one timing in `timings`
- a valid `firstTiming.begin`
- a location with:
  - `city`
  - `name`
  - `address`
  - `latitude`
  - `longitude`

### Drop

Drop rows only when they are malformed or unusable for the RAG POC.

Examples:

- missing title
- missing short description
- missing first timing
- missing venue name
- missing address
- missing coordinates
- missing city

### Do not do primary filtering here

The raw API fetch is responsible for:

- Paris geographic scope
- one-year minimum timing bound
- inclusion of future events

The preprocessing script may log or count suspicious records, but it should not silently change the project filtering logic.

---

## Validation expectations

Later tests should verify:

- processed dataset exists
- processed dataset is not empty
- all rows have required fields
- all rows have `city = Paris`
- all rows have valid coordinates
- all rows have `first_timing_begin >= today - 365 days`
- all rows have usable `text_for_embedding`
