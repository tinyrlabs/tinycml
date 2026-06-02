# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in TinyCML, please report it through one of the following channels:

- **GitHub Issues:** Open an issue and apply the `security` label.
- **Email:** Contact the maintainer directly via the email listed on the GitHub profile.

Please include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce the issue.
- The affected version(s).
- Any suggested fixes, if available.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| main (latest) | Yes |
| Previous releases | Best-effort |

Security fixes are applied to the `main` branch. Releases are tagged periodically from `main`.

## Response Timeline

- **Acknowledgment:** Within 48 hours of receiving a report.
- **Critical vulnerabilities:** A fix will be targeted within 7 days. Critical issues include memory safety bugs (buffer overflows, use-after-free, etc.) that can be triggered through normal library usage.
- **Non-critical vulnerabilities:** Fixes will be included in the next regular release cycle.

## Scope

TinyCML is a pure C library with no external dependencies. Security concerns relevant to this project include:

- Buffer overflows or out-of-bounds memory access.
- Integer overflows leading to incorrect allocations.
- Use-after-free or double-free errors.
- Null pointer dereferences in public APIs.

Issues in downstream applications that misuse the library API are generally outside the scope of this policy, but we still welcome reports if the library could reasonably prevent the misuse.
