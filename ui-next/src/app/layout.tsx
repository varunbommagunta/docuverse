import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DocuVerse — Document Q&A",
  description: "Hybrid RAG · PDF Q&A · Cited Answers",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden bg-white text-gray-900 font-sans text-sm">
        {children}
      </body>
    </html>
  );
}
