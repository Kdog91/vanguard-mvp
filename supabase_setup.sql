-- Vanguard Analytics, update 27: run once in Supabase -> SQL Editor
create table if not exists public.projects (
  id           uuid primary key default gen_random_uuid(),
  owner        text not null check (owner ~ '^[0-9a-f]{64}$'),
  name         text not null check (char_length(name) <= 80),
  source       text,
  n_rows       integer,
  n_cols       integer,
  size_bytes   bigint,
  storage_path text not null,
  settings     jsonb not null default '{}'::jsonb,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  expires_at   timestamptz not null default now() + interval '180 days'
);
create index if not exists projects_owner_updated on public.projects (owner, updated_at desc);
create index if not exists projects_expires on public.projects (expires_at);

-- Locked down: row level security on, no policies, nothing granted to the public roles.
alter table public.projects enable row level security;
revoke all on public.projects from anon, authenticated;
grant select, insert, update, delete on public.projects to service_role;

-- Private bucket for the encrypted project files (50 MB per file).
insert into storage.buckets (id, name, public, file_size_limit)
values ('projects', 'projects', false, 52428800)
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;
