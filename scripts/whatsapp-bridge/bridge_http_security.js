import { timingSafeEqual } from 'node:crypto';

/**
 * Authenticate the private Python↔Node bridge channel.
 *
 * Every bridge process must have a high-entropy token. Pair-only launches use
 * an ephemeral token because they do not expose the HTTP API to a client.
 */
export function hasValidBridgeToken(expectedToken, authorizationHeader) {
  const expected = String(expectedToken || '');
  if (!expected) return false;

  const header = String(authorizationHeader || '');
  if (!header.startsWith('Bearer ')) return false;

  const supplied = Buffer.from(header.slice('Bearer '.length), 'utf8');
  const expectedBytes = Buffer.from(expected, 'utf8');
  if (supplied.length !== expectedBytes.length) return false;

  return timingSafeEqual(supplied, expectedBytes);
}
