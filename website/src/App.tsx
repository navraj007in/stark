import { CodeBlock } from './components/CodeBlock';
import { Mark } from './components/Mark';
import {
  REPO,
  conformance,
  coreExample,
  example,
  hero,
  install,
  links,
  pillars,
  problem,
  state,
} from './content';

export default function App() {
  return (
    <>
      <a className="skip" href="#main">Skip to content</a>

      <header className="site-header">
        <div className="wrap site-header__inner">
          <a className="brand" href="#top" aria-label="STARK home">
            <Mark size={30} />
            <span className="brand__word">STARK</span>
          </a>
          <nav className="nav" aria-label="Primary">
            <a href="#why">Why</a>
            <a href="#language">Language</a>
            <a href="#conformance">Conformance</a>
            <a href="#install">Install</a>
            <a href="#status">Status</a>
            <a className="nav__cta" href={REPO} rel="noreferrer noopener">GitHub</a>
          </nav>
        </div>
      </header>

      <main id="main">
        {/* ---------------------------------------------------------------- hero -- */}
        <section className="hero" id="top">
          <div className="wrap hero__inner">
            <p className="badge">{hero.status}</p>
            <h1 className="hero__title">{hero.title}</h1>
            <p className="hero__lede">{hero.lede}</p>
            <div className="hero__actions">
              <a className="btn btn--primary" href="#install">Get started</a>
              <a className="btn" href={REPO} rel="noreferrer noopener">Read the source</a>
            </div>
            <CodeBlock code={example.code} caption={example.caption} />
          </div>
        </section>

        {/* ------------------------------------------------------------- problem -- */}
        <section className="section" id="why">
          <div className="wrap">
            <h2 className="section__title">{problem.title}</h2>
            <p className="section__lede">{problem.body}</p>
            <ul className="failures">
              {problem.items.map((item) => (
                <li key={item} className="failures__item">{item}</li>
              ))}
            </ul>
            <p className="section__after">
              Each is a mismatch between two declarations that were never compared. STARK compares
              them.
            </p>
          </div>
        </section>

        {/* ------------------------------------------------------------- pillars -- */}
        <section className="section section--alt" id="language">
          <div className="wrap">
            <h2 className="section__title">What the language guarantees</h2>
            <div className="grid">
              {pillars.map((p) => (
                <article key={p.title} className="card">
                  <h3 className="card__title">{p.title}</h3>
                  <p className="card__body">{p.body}</p>
                </article>
              ))}
            </div>
            <CodeBlock code={coreExample.code} caption={coreExample.caption} />
          </div>
        </section>

        {/* --------------------------------------------------------- conformance -- */}
        <section className="section conformance" id="conformance">
          <div className="wrap">
            <h2 className="section__title">{conformance.title}</h2>
            <p className="section__lede">{conformance.body}</p>

            <div className="engines" role="list" aria-label="Execution engines">
              {['Reference interpreter', 'MIR interpreter', 'Native · debug', 'Native · release'].map(
                (engine) => (
                  <div className="engine" role="listitem" key={engine}>
                    <span className="engine__dot" aria-hidden="true" />
                    {engine}
                  </div>
                ),
              )}
            </div>

            <blockquote className="pull">{conformance.emphasis}</blockquote>
            <p className="section__after">{conformance.note}</p>
          </div>
        </section>

        {/* ------------------------------------------------------------- install -- */}
        <section className="section section--alt" id="install">
          <div className="wrap">
            <h2 className="section__title">{install.title}</h2>
            <ol className="steps">
              {install.steps.map((step, i) => (
                <li key={step.label} className="step">
                  <div className="step__head">
                    <span className="step__num" aria-hidden="true">{i + 1}</span>
                    <h3 className="step__label">{step.label}</h3>
                  </div>
                  <CodeBlock code={step.code} language="shell" />
                </li>
              ))}
            </ol>
            <p className="section__after">{install.footnote}</p>
          </div>
        </section>

        {/* -------------------------------------------------------------- status -- */}
        <section className="section" id="status">
          <div className="wrap">
            <h2 className="section__title">{state.title}</h2>
            <p className="section__lede">{state.lede}</p>
            <div className="status">
              <div className="status__col">
                <h3 className="status__head status__head--yes">Working today</h3>
                <ul className="status__list">
                  {state.working.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
              <div className="status__col">
                <h3 className="status__head status__head--no">Not yet, or deliberately not</h3>
                <ul className="status__list">
                  {state.notYet.map((item) => <li key={item}>{item}</li>)}
                </ul>
              </div>
            </div>
            <p className="section__after">
              STARK is a research compiler with a working implementation, not a production language.
              The status file in the repository is the authority; this page summarises it and can lag.
            </p>
          </div>
        </section>
      </main>

      <footer className="site-footer">
        <div className="wrap site-footer__inner">
          <div className="site-footer__brand">
            <Mark size={26} />
            <div>
              <p className="site-footer__word">STARK</p>
              <p className="site-footer__note">MIT licensed · pre-alpha</p>
            </div>
          </div>
          <ul className="site-footer__links">
            {links.map((l) => (
              <li key={l.label}>
                <a href={l.href} rel="noreferrer noopener">{l.label}</a>
                <span>{l.note}</span>
              </li>
            ))}
          </ul>
        </div>
      </footer>
    </>
  );
}
