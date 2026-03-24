# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a Vulnerability

**Please do not open public GitHub issues for security vulnerabilities.**

Report security issues privately to: **darshjme@gmail.com**

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge your report within **48 hours** and aim to release a patch within **7 days** for critical issues.

## Security Model

`agent-guardrails` is a **local, offline library**. It does not make network requests, store data, or call external APIs. All processing happens in-process.

### Pattern-Based Detection Limitations

`ContentGuard` uses regex patterns for PII and content detection. Be aware:

- **False positives** are possible (e.g. random numbers matching credit card patterns)
- **False negatives** are possible (novel obfuscation techniques may evade patterns)
- This library is **not a substitute** for a dedicated content moderation system in high-risk applications

### Dependency Security

This library depends only on `pydantic` (validation) and Python stdlib. Keep pydantic updated to receive upstream security patches:

```bash
pip install --upgrade pydantic
```
