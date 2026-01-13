# YouTube Uploads

The pipeline can optionally upload the final merged video to YouTube after the Telegram send completes.

## Required Environment Variables

Set these in Railway Variables if you want uploads enabled:

- `YT_UPLOAD_ON_COMPLETE=1` to enable uploads (default: disabled).
- `YT_CLIENT_SECRETS_B64` base64-encoded `client_secrets.json`.
- `YT_TOKEN_B64` base64-encoded OAuth token JSON.

The runtime decodes these values to:

- `/tmp/yt/client_secrets.json`
- `/tmp/yt/token.json`

If either value is missing, the upload step is skipped and the job continues.

## Optional Settings

- `YT_SCHEDULE_DELAY_MINUTES` (default: `60`) — publish time offset.
- `YT_TIMEZONE` (default: `Asia/Kolkata`) — date formatting only.
- `YT_CATEGORY` (default: `22`).
- `YT_KEYWORDS` — comma-separated tags.
- `YT_DESCRIPTION_TEMPLATE` — description override with `{DATE}` placeholder.

## Notes

Uploads are scheduled for `now_utc + YT_SCHEDULE_DELAY_MINUTES`, with `privacyStatus` set to `private` and `publishAt` set to the RFC3339 UTC timestamp.
