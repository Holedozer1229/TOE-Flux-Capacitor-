# Save CONTRIBUTING.md
cat << 'EOF' > CONTRIBUTING.md
# Contributing to TOE Flux Capacitor

We welcome contributions to advance quantum-gravity research! Please follow these steps:

1. Fork the repository.
2. Create a branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to your fork (`git push origin feature/your-feature`).
5. Open a pull request.

Ensure tests pass (`pytest tests/`) and follow PEP 8 style.
EOF

# Commit and push
git add CONTRIBUTING.md
git commit -m "Add CONTRIBUTING.md with guidelines"
git push origin main
