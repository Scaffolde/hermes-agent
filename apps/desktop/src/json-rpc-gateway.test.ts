import { JsonRpcGatewayClient } from '@hermes/shared'
import { describe, expect, it, vi } from 'vitest'

type Listener = (event?: unknown) => void

class FakeWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSED = 3

  readyState = FakeWebSocket.CONNECTING
  closed = false
  sent: string[] = []
  private listeners: Record<string, Set<Listener>> = {}

  addEventListener(type: string, fn: Listener) {
    ;(this.listeners[type] ??= new Set()).add(fn)
  }

  removeEventListener(type: string, fn: Listener) {
    this.listeners[type]?.delete(fn)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.closed = true
    this.readyState = FakeWebSocket.CLOSED
    // Real WebSocket close events are asynchronous. Do not emit here: this
    // catches callers that clear their socket reference before doing their own
    // closed-state/pending-request cleanup.
  }

  open() {
    this.readyState = FakeWebSocket.OPEN
    this.emit('open')
  }

  emit(type: string, event?: unknown) {
    for (const fn of this.listeners[type] ?? []) {
      fn(event)
    }
  }
}

describe('JsonRpcGatewayClient', () => {
  it('close() immediately transitions closed and rejects pending requests', async () => {
    vi.stubGlobal('WebSocket', FakeWebSocket)
    const socket = new FakeWebSocket()

    const client = new JsonRpcGatewayClient({
      closedErrorMessage: 'closed for test',
      socketFactory: () => socket as unknown as WebSocket
    })

    const states: string[] = []
    client.onState(state => states.push(state))

    const connect = client.connect('ws://localhost/api/ws')
    socket.open()
    await connect

    const pending = client.request('session.info')
    client.close()

    expect(socket.closed).toBe(true)
    expect(client.connectionState).toBe('closed')
    await expect(pending).rejects.toThrow('closed for test')
    expect(states).toEqual(['idle', 'connecting', 'open', 'closed'])

    // The eventual close event from the old socket must be harmless: close()
    // already cleaned up and rejected pending work.
    socket.emit('close')
    expect(client.connectionState).toBe('closed')
  })
})
