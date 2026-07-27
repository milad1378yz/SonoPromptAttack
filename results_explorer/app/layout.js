export const metadata = {
  title: "SonoPromptAttack Result Explorer",
  description:
    "Explore recorded prompt attacks by proposer LLM, target MedVLM, task, and example.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0 }}>{children}</body>
    </html>
  );
}
