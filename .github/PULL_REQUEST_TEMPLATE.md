## Description

<!-- Provide a clear description of your changes -->

## Type of Change

<!-- Mark the appropriate option with an [x] -->

- [ ] ✨ New feature (`feat`)
- [ ] 🐛 Bug fix (`fix`)
- [ ] 📝 Documentation (`docs`)
- [ ] ♻️ Refactoring (`refactor`)
- [ ] ⚡ Performance improvement (`perf`)
- [ ] ✅ Tests (`test`)
- [ ] 🔧 Configuration (`chore`)
- [ ] 🏗️ Build/CI changes (`ci`)

## Related Issues

<!-- Link any related issues using GitHub syntax, e.g., "Closes #123" or "Related to #456" -->

## Testing

<!-- Describe the testing you performed -->

### Test Coverage
- [ ] Unit tests added/updated
- [ ] Integration tests verified
- [ ] Manual testing completed

### Test Details
<!-- Describe specific tests performed -->


## Checklist

<!-- Mark items as completed [x] -->

### Code Quality
- [ ] Code follows project style guidelines (Ruff)
- [ ] Self-review of code completed
- [ ] Code is properly commented where necessary
- [ ] No new warnings introduced
- [ ] Linting passes: `pixi run lint`
- [ ] Formatting passes: `pixi run format`

### Testing
- [ ] Tests pass locally: `pixi run test`
- [ ] Pipeline validation successful
- [ ] Backward compatibility maintained

### Documentation
- [ ] README updated if needed
- [ ] Feature documentation added/updated in `docs/`
- [ ] Commit messages follow Conventional Commits standard
- [ ] Changelog entry added (if applicable)

### Security
- [ ] No sensitive data committed (API keys, passwords)
- [ ] Environment variables used appropriately
- [ ] No unnecessary dependencies added

## Screenshots / Recordings

<!-- If applicable, add screenshots or recordings to demonstrate changes -->

## Additional Notes

<!-- Any other context, concerns, or questions for reviewers -->

---

## For Reviewers

### Review Checklist
- [ ] Code changes are appropriate for the branch target
- [ ] Logic and algorithms are correct
- [ ] Error handling is adequate
- [ ] Performance implications considered
- [ ] Security implications reviewed
- [ ] Documentation is complete

### Suggested Approval Criteria
- **LGTM**: Approved without reservations
- **Approved with minor comments**: Non-blocking suggestions
- **Changes requested**: Blocking issues must be addressed

---

**Branch:** `${{ github.head_ref }}` → **Target:** `${{ github.base_ref }}`  
**Commit:** `${{ github.sha }}`
