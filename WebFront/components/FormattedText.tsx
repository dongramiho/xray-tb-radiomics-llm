import styles from "./FormattedText.module.css";

const BULLET_RE = /^\s*[-•*]\s+/;

type TextGroup = { type: "ul" | "p"; lines: string[] };

// Plain-text formatting without a markdown library: blank lines split paragraphs,
// consecutive "- " / "• " lines become a bullet list, single line breaks are kept.
// Shared by the chat bubble (clinician text) and the analysis card (patient-friendly text).
export default function FormattedText({ text }: { text: string }) {
  const blocks = text.split(/\n[ \t]*\n/);
  return (
    <div className={styles.blocks}>
      {blocks.map((block, blockIdx) => {
        const groups: TextGroup[] = [];
        for (const line of block.split("\n")) {
          if (line.trim() === "") continue;
          const type: TextGroup["type"] = BULLET_RE.test(line) ? "ul" : "p";
          const last = groups[groups.length - 1];
          if (last && last.type === type) last.lines.push(line);
          else groups.push({ type, lines: [line] });
        }
        return groups.map((group, groupIdx) =>
          group.type === "ul" ? (
            <ul key={`${blockIdx}-${groupIdx}`} className={styles.bulletList}>
              {group.lines.map((line, lineIdx) => (
                <li key={lineIdx}>{line.replace(BULLET_RE, "")}</li>
              ))}
            </ul>
          ) : (
            <p key={`${blockIdx}-${groupIdx}`} className={styles.paragraph}>
              {group.lines.join("\n")}
            </p>
          ),
        );
      })}
    </div>
  );
}
