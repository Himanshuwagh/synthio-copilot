-- Run in Supabase Dashboard → SQL Editor → New query

create table if not exists public.chat_messages (
  id         uuid primary key default gen_random_uuid(),
  session_id text not null,
  role       text not null check (role in ('user', 'assistant')),
  message    text not null,
  created_at timestamptz not null default now()
);

create index if not exists chat_messages_session_created_idx
  on public.chat_messages (session_id, created_at);

comment on table public.chat_messages is
  'Pharma co-pilot chat turns; scoped by browser session_id.';
