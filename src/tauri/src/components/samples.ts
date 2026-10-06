import type { Msg } from "./Hero.js";

/**
 * Sample threads for UI work: `/sample [name]` appends one to the
 * transcript. Add a new named thread here for each UI you want to
 * style — no backend needed to preview it.
 *
 * 1st: "chain" — chain of action. Mirrors the voice agent's web-search
 * flow: search results, a found profile photo, a conclusion, and one
 * still-running search, followed by the agent's reply.
 */
const CHAIN: Msg[] = [
  {
    id: "s-you-1",
    who: "you",
    text: "Who is Hayden Bleasel and what has he worked on recently?",
  },
  {
    id: "s-cot-1",
    who: "cot",
    text: "chain of thought",
    steps: [
      {
        kind: "search",
        label: "Searching for profiles for Hayden Bleasel",
        status: "complete",
        results: ["x.com", "instagram.com", "github.com"],
      },
      {
        kind: "image",
        label: "Found the profile photo for Hayden Bleasel",
        status: "complete",
        src: "https://avatar.vercel.sh/haydenbleasel",
        alt: "Example profile",
        caption: "Hayden Bleasel's profile photo from x.com, showing a Ghibli-style man.",
      },
      {
        kind: "text",
        label:
          "Hayden Bleasel is an Australian product designer, software engineer, and founder. He is currently based in the United States working for Vercel, an American cloud application company.",
        status: "complete",
      },
      {
        kind: "search",
        label: "Searching for recent work...",
        status: "active",
        results: ["github.com", "dribbble.com"],
      },
    ],
  },
  {
    id: "s-agent-1",
    who: "agent",
    text: "Hayden Bleasel is an Australian product designer and engineer, now in the US at Vercel. Found profiles on x.com, instagram.com and github.com. Still scanning github.com and dribbble.com for recent work — want me to keep digging?",
  },
];

const CODE: Msg[] = [
  {
    id: "c-you-1",
    who: "you",
    text: "Can you write a React greeting component for me?",
  },
  {
    id: "c-agent-1",
    who: "agent",
    text: "Here's a small greeting component you can drop in:\n```jsx\nfunction MyComponent(props) {\n  return (\n    <div>\n      <h1>Hello, {props.name}!</h1>\n      <p>This is an example React component.</p>\n    </div>\n  );\n}\n```\nPass `name` as a prop to customize who it greets.",
  },
];

const TURN: Msg[] = [
  {
    id: "t-you-1",
    who: "you",
    text: "How big is everything here, and did I leave any secrets behind?",
  },
  {
    id: "t-sys-1",
    who: "sys",
    text: "Measuring disk usage first, then grepping for leaked secrets.",
  },
  {
    id: "t-sys-2",
    who: "sys",
    text: 'tool: shell {"action":"shell","command":"du -sh img docs src"}',
  },
  {
    id: "t-sys-3",
    who: "sys",
    text: "img 12M, docs 3M, src 8M",
  },
  {
    id: "t-sys-4",
    who: "sys",
    text: 'tool: shell {"action":"shell","command":"grep -ri key|token|secret ."}',
  },
  {
    id: "t-sys-5",
    who: "sys",
    text: "no matches",
  },
  {
    id: "t-agent-1",
    who: "agent",
    text: "23M total — images are the bulk of it. Clean — no secrets or keys found.",
  },
];

/** Whole sample conversation: every thread back to back. Shown as the pinned "sample" session. */
export const SAMPLE_CONVERSATION: Msg[] = [...CHAIN, ...CODE, ...TURN];

export const SAMPLES: Record<string, Msg[]> = { chain: CHAIN, code: CODE, turn: TURN };

export const SAMPLE_NAMES = Object.keys(SAMPLES);
