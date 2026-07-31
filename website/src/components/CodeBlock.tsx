import { useMemo } from 'react';
import { tokenize } from '../stark-highlight';

interface Props {
  code: string;
  caption?: string;
  /** Shell snippets skip STARK tokenising — highlighting them as STARK would be nonsense. */
  language?: 'stark' | 'shell';
}

/**
 * A code sample.
 *
 * Rendered as `<pre><code>` with token spans rather than `dangerouslySetInnerHTML`: the samples are
 * static and trusted, but building the DOM from tokens means no HTML-escaping bug is possible here
 * even if a sample later contains `<` or `&`.
 */
export function CodeBlock({ code, caption, language = 'stark' }: Props) {
  const tokens = useMemo(
    () => (language === 'stark' ? tokenize(code) : null),
    [code, language],
  );

  return (
    <figure className="code">
      {caption && <figcaption className="code__caption">{caption}</figcaption>}
      <pre className="code__pre">
        <code>
          {tokens
            ? tokens.map((token, index) => (
                <span key={index} className={`tok tok--${token.kind}`}>
                  {token.text}
                </span>
              ))
            : code}
        </code>
      </pre>
    </figure>
  );
}
