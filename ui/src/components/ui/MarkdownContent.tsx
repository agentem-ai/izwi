import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface MarkdownContentProps {
  content: string;
  className?: string;
}

function isExternalLink(href?: string): boolean {
  if (!href) {
    return false;
  }
  return /^(https?:)?\/\//i.test(href);
}

export function MarkdownContent({ content, className }: MarkdownContentProps) {
  return (
    <div className={clsx("chat-markdown", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children, node: _node, ...props }) => {
            const external = isExternalLink(href);
            return (
              <a
                {...props}
                href={href}
                target={external ? "_blank" : undefined}
                rel={external ? "noopener noreferrer nofollow" : undefined}
              >
                {children}
              </a>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
