import { atom } from 'nanostores'

import { capitalize } from '@/lib/text'

export type SubagentStatus = 'completed' | 'error' | 'failed' | 'interrupted' | 'queued' | 'running' | 'timeout'
export type SubagentStreamKind = 'progress' | 'summary' | 'thinking' | 'tool'

export interface SubagentStreamEntry {
  at: number
  isError?: boolean
  kind: SubagentStreamKind
  text: string
}

export interface SubagentProgress {
  id: string
  parentId: null | string
  goal: string
  /** The child's own stored session id — lets UIs open its session window. */
  sessionId?: string
  model?: string
  status: SubagentStatus
  taskCount: number
  taskIndex: number
  startedAt: number
  updatedAt: number
  durationSeconds?: number
  costUsd?: number
  inputTokens?: number
  outputTokens?: number
  toolCount?: number
  filesRead: string[]
  filesWritten: string[]
  stream: SubagentStreamEntry[]
  summary?: string
  /** Active tool while running — cleared on terminal status. */
  currentTool?: string
  /** Synthetic row created from delegation.status polling. */
  source?: 'delegation-status'
}

export interface SubagentNode extends SubagentProgress {
  children: SubagentNode[]
}

export type SubagentPayload = Record<string, unknown>

const TERMINAL: ReadonlySet<SubagentStatus> = new Set(['completed', 'error', 'failed', 'interrupted', 'timeout'])
const FAILED: ReadonlySet<SubagentStatus> = new Set(['error', 'failed', 'interrupted', 'timeout'])
export const DELEGATION_STATUS_SESSION_ID = '__delegation_status__'
const MAX_STREAM = 24
const PREVIEW_MAX = 220
const TOOL_PREVIEW_MAX = 96

export const $subagentsBySession = atom<Record<string, SubagentProgress[]>>({})

const isStr = (v: unknown): v is string => typeof v === 'string'
const str = (v: unknown) => (isStr(v) ? v : '')
const num = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
const strList = (v: unknown) => (Array.isArray(v) ? v.filter(isStr) : [])

const asStatus = (v: unknown): SubagentStatus => {
  if (
    v === 'completed' ||
    v === 'error' ||
    v === 'failed' ||
    v === 'interrupted' ||
    v === 'queued' ||
    v === 'running' ||
    v === 'timeout'
  ) {
    return v
  }

  return 'running'
}

const epochMs = (v: unknown) => {
  const n = num(v)

  if (n == null) {
    return undefined
  }

  return n < 10_000_000_000 ? n * 1000 : n
}

const compact = (text: string, max = PREVIEW_MAX) => {
  const line = text.replace(/\s+/g, ' ').trim()

  if (!line) {
    return ''
  }

  return line.length > max ? `${line.slice(0, max - 1)}…` : line
}

const toolLabel = (name: string) => name.split('_').filter(Boolean).map(capitalize).join(' ') || name

const formatTool = (name: string, preview = '') => {
  const snippet = compact(preview, TOOL_PREVIEW_MAX)

  return snippet ? `${toolLabel(name)}("${snippet}")` : toolLabel(name)
}

interface TailEntry {
  isError?: boolean
  preview?: string
  tool?: string
}

const asTail = (v: unknown): TailEntry[] =>
  Array.isArray(v)
    ? v
        .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
        .map(item => ({
          isError: item.is_error === true,
          preview: str(item.preview) || undefined,
          tool: str(item.tool) || undefined
        }))
    : []

const idOf = (p: SubagentPayload) =>
  str(p.subagent_id) || `${str(p.parent_id) || 'root'}:${num(p.task_index) ?? 0}:${str(p.goal)}`

const appendStream = (stream: SubagentStreamEntry[], entry: SubagentStreamEntry) => {
  const last = stream.at(-1)

  if (last?.kind === entry.kind && last.text === entry.text && last.isError === entry.isError) {
    return stream
  }

  return [...stream, entry].slice(-MAX_STREAM)
}

function streamFromPayload(
  payload: SubagentPayload,
  status: SubagentStatus,
  eventType: string,
  at: number
): SubagentStreamEntry[] {
  const out: SubagentStreamEntry[] = []
  const tool = str(payload.tool_name)
  const preview = str(payload.tool_preview) || str(payload.text)
  const text = compact(str(payload.text) || preview)

  for (const tail of asTail(payload.output_tail)) {
    const line = tail.tool ? formatTool(tail.tool, tail.preview ?? '') : compact(tail.preview ?? '')

    if (line) {
      out.push({ at, isError: tail.isError, kind: tail.tool ? 'tool' : 'progress', text: line })
    }
  }

  if (tool) {
    out.push({ at, isError: !!payload.error, kind: 'tool', text: formatTool(tool, preview) })
  }

  if (eventType === 'subagent.progress' && text) {
    out.push({ at, isError: !!payload.error, kind: 'progress', text })
  }

  if (eventType === 'subagent.thinking' && text) {
    out.push({ at, kind: 'thinking', text })
  }

  const summary = compact(str(payload.summary) || str(payload.text))

  if (TERMINAL.has(status) && summary) {
    out.push({ at, isError: FAILED.has(status), kind: 'summary', text: summary })
  }

  return out
}

function toProgress(payload: SubagentPayload, prev: SubagentProgress | undefined, eventType = ''): SubagentProgress {
  const at = Date.now()
  const status = asStatus(payload.status)
  const tool = str(payload.tool_name)
  const stream = streamFromPayload(payload, status, eventType, at).reduce(appendStream, prev?.stream ?? [])
  const filesRead = strList(payload.files_read)
  const filesWritten = strList(payload.files_written)

  return {
    id: prev?.id ?? idOf(payload),
    parentId: str(payload.parent_id) || prev?.parentId || null,
    goal: str(payload.goal) || prev?.goal || 'Subagent',
    sessionId: str(payload.child_session_id) || prev?.sessionId,
    model: str(payload.model) || prev?.model,
    status,
    taskCount: num(payload.task_count) ?? prev?.taskCount ?? 1,
    taskIndex: num(payload.task_index) ?? prev?.taskIndex ?? 0,
    startedAt: prev?.startedAt ?? epochMs(payload.started_at) ?? epochMs(payload.dispatched_at) ?? at,
    updatedAt: at,
    durationSeconds: num(payload.duration_seconds) ?? prev?.durationSeconds,
    costUsd: num(payload.cost_usd) ?? prev?.costUsd,
    inputTokens: num(payload.input_tokens) ?? prev?.inputTokens,
    outputTokens: num(payload.output_tokens) ?? prev?.outputTokens,
    toolCount: num(payload.tool_count) ?? prev?.toolCount,
    filesRead: filesRead.length ? filesRead : (prev?.filesRead ?? []),
    filesWritten: filesWritten.length ? filesWritten : (prev?.filesWritten ?? []),
    stream,
    summary: str(payload.summary) || prev?.summary,
    currentTool: TERMINAL.has(status) ? undefined : tool || prev?.currentTool,
    source: payload.source === 'delegation-status' ? 'delegation-status' : prev?.source
  }
}

export function clearSessionSubagents(sid: string) {
  const map = $subagentsBySession.get()

  if (!(sid in map)) {
    return
  }

  const { [sid]: _drop, ...rest } = map
  $subagentsBySession.set(rest)
}

export function pruneDelegateFallbackSubagents(sid: string) {
  const map = $subagentsBySession.get()
  const list = map[sid]

  if (!list?.length) {
    return
  }

  const next = list.filter(item => !item.id.startsWith('delegate-tool:'))

  if (next.length === list.length) {
    return
  }

  $subagentsBySession.set({ ...map, [sid]: next })
}

export function upsertSubagent(sid: string, payload: SubagentPayload, createIfMissing = true, eventType?: string) {
  const map = $subagentsBySession.get()
  const list = map[sid] ?? []
  const id = idOf(payload)
  const idx = list.findIndex(item => item.id === id)

  if (idx < 0 && !createIfMissing) {
    return
  }

  const prev = idx >= 0 ? list[idx] : undefined

  if (prev && TERMINAL.has(prev.status)) {
    return
  }

  const next = toProgress(payload, prev, eventType)
  const nextList = idx >= 0 ? list.map(item => (item.id === id ? next : item)) : [...list, next]

  $subagentsBySession.set({ ...map, [sid]: nextList })
}

export function buildSubagentTree(items: readonly SubagentProgress[]): SubagentNode[] {
  const nodes = new Map<string, SubagentNode>()

  for (const item of items) {
    nodes.set(item.id, { ...item, children: [] })
  }

  const roots: SubagentNode[] = []

  for (const node of nodes.values()) {
    const parent = node.parentId ? nodes.get(node.parentId) : null

    if (parent) {
      parent.children.push(node)
    } else {
      roots.push(node)
    }
  }

  const sort = (a: SubagentNode, b: SubagentNode) =>
    a.startedAt - b.startedAt || a.taskIndex - b.taskIndex || a.goal.localeCompare(b.goal)

  const walk = (node: SubagentNode) => node.children.sort(sort).forEach(walk)
  roots.sort(sort).forEach(walk)

  return roots
}

export const activeSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => item.status === 'queued' || item.status === 'running').length

export const failedSubagentCount = (items: readonly SubagentProgress[]) =>
  items.filter(item => FAILED.has(item.status)).length

/** Flatten every session's subagents — the scope the Spawn-tree panel and the
 *  status-bar indicator must agree on. */
export const allSubagents = (bySession: Record<string, SubagentProgress[]>) => {
  const byId = new Map<string, SubagentProgress>()

  const prefer = (current: SubagentProgress, candidate: SubagentProgress) => {
    const currentActive = current.status === 'queued' || current.status === 'running'
    const candidateActive = candidate.status === 'queued' || candidate.status === 'running'

    if (candidateActive !== currentActive) {
      return candidateActive ? candidate : current
    }

    if (current.source === 'delegation-status' && candidate.source !== 'delegation-status') {
      return candidate
    }

    if (candidate.source === 'delegation-status' && current.source !== 'delegation-status') {
      return current
    }

    return candidate.updatedAt >= current.updatedAt ? candidate : current
  }

  for (const items of Object.values(bySession)) {
    for (const item of items) {
      const current = byId.get(item.id)
      byId.set(item.id, current ? prefer(current, item) : item)
    }
  }

  return [...byId.values()]
}

/**
 * Mirror backend delegation.status active records into the desktop store.
 *
 * Native subagent.* events are still the rich source of truth, but they can be
 * silent while a child is inside a long tool call, and a new parent turn clears
 * the per-session live rows. delegation.status is the backend liveness registry;
 * keeping it in a synthetic session prevents the Agents status item and Spawn
 * tree from claiming zero while child agents are still actually running.
 */
export function syncActiveDelegationSubagents(active: readonly Record<string, unknown>[]) {
  const map = $subagentsBySession.get()
  const cleaned: Record<string, SubagentProgress[]> = {}
  const groups: Record<string, SubagentProgress[]> = {}

  for (const [sid, items] of Object.entries(map)) {
    const next = items.filter(item => item.source !== 'delegation-status')

    if (next.length) {
      cleaned[sid] = next
    }
  }

  for (const [index, item] of active.filter(item => str(item.subagent_id)).entries()) {
    const sid = str(item.parent_session_id) || DELEGATION_STATUS_SESSION_ID
    const previous = (map[sid] ?? []).find(prev => prev.id === str(item.subagent_id))

    const progress = toProgress(
      {
        ...item,
        source: 'delegation-status',
        status: item.status ?? 'running',
        task_index: num(item.task_index) ?? index
      },
      previous,
      'delegation.status'
    )

    groups[sid] = [...(groups[sid] ?? []), progress]
  }

  if (!Object.keys(groups).length) {
    const hadSyntheticRows = Object.values(map).some(items => items.some(item => item.source === 'delegation-status'))

    if (hadSyntheticRows) {
      $subagentsBySession.set(cleaned)
    }

    return
  }

  $subagentsBySession.set(
    Object.fromEntries(
      Object.entries({ ...cleaned, ...groups }).map(([sid, items]) => [
        sid,
        sid in groups ? [...(cleaned[sid] ?? []), ...items] : items
      ])
    )
  )
}
