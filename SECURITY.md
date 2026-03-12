# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 12.x    | Yes                |
| < 12.0  | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability in Fracttalix, please report it
responsibly:

1. **Do not** open a public GitHub issue.
2. Email the maintainer at the address listed in `pyproject.toml`, or use
   [GitHub's private vulnerability reporting](https://github.com/thomasbrennan/Fracttalix/security/advisories/new).
3. Include a description of the vulnerability, steps to reproduce, and any
   relevant environment details.

You can expect an initial response within 7 days. Critical issues will be
patched and released as soon as possible.

## Scope

Fracttalix has zero required runtime dependencies and is pure Python (stdlib
only in core). The primary attack surface is:

- Malicious input to the streaming detector API
- The optional REST server (`fracttalix.extras.server`)
- Dependencies in optional extras (`fast`, `full`)

The optional REST server is intended for local/development use and should not
be exposed to untrusted networks without appropriate access controls.
