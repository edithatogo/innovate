// Package innovate provides a thin Go binding over the shared innovate kernel.
//
// The first phase keeps the surface intentionally minimal: define a stable
// package root, a bridge path, and kernel request/response envelope types that
// mirror the language-neutral contract exposed by src/innovate/kernel.py.
package innovate
