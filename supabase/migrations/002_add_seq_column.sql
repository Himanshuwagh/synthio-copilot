-- Run in Supabase Dashboard → SQL Editor → New query
-- Adds a monotonically-increasing sequence column so message order is always
-- stable regardless of created_at ties (e.g. rows saved with the same timestamp).

alter table public.chat_messages
  add column if not exists seq bigint generated always as identity;

-- Index to make session-ordered queries fast
create index if not exists chat_messages_session_seq_idx
  on public.chat_messages (session_id, seq);
