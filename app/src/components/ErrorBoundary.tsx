import { Component, type ReactNode } from 'react'

/**
 * One bad panel must never take down the terminal.
 *
 * This exists because it already happened: the Signals panel called
 * .toUpperCase() on a numeric field, React unmounted the entire tree, and every
 * other panel went blank with it. A terminal whose calendar disappears because
 * the signals payload changed shape is not usable.
 */
export default class ErrorBoundary extends Component<
  { children: ReactNode; label: string },
  { err: Error | null }
> {
  state: { err: Error | null } = { err: null }

  static getDerivedStateFromError(err: Error) { return { err } }

  componentDidCatch(err: Error) {
    console.error('[panel crashed]', this.props.label, err)
  }

  render() {
    if (!this.state.err) return this.props.children
    return (
      <div className="h-full flex flex-col items-center justify-center gap-2 p-6 text-center">
        <p className="text-[12px] text-down">The {this.props.label} panel failed to render.</p>
        <p className="text-[11px] text-faint max-w-md font-mono">{this.state.err.message}</p>
        <button
          onClick={() => this.setState({ err: null })}
          className="mt-1 px-3 py-1 rounded bg-raised text-[11px] text-muted hover:text-ink"
        >Retry</button>
      </div>
    )
  }
}
