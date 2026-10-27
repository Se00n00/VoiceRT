/**
 * Connector shelf under the composer: the remote MCP apps the backend can
 * serve. Icons are brand marks from gilbarbara/logos (CC0). Toggles are
 * local until the bridge exposes a per-connector enable endpoint.
 */
import { For, Show, createSignal } from "solid-js";
import asana from "../assets/connectors/asana.svg";
import box from "../assets/connectors/box.svg";
import calcom from "../assets/connectors/calcom.svg";
import dropbox from "../assets/connectors/dropbox.svg";
import github from "../assets/connectors/github.svg";
import gmail from "../assets/connectors/gmail.svg";
import googleCalendar from "../assets/connectors/google-calendar.svg";
import googleDrive from "../assets/connectors/google-drive.svg";
import jira from "../assets/connectors/jira.svg";
import linear from "../assets/connectors/linear.svg";
import microsoftTeams from "../assets/connectors/microsoft-teams.svg";
import salesforce from "../assets/connectors/salesforce.svg";
import slack from "../assets/connectors/slack.svg";
import stripe from "../assets/connectors/stripe.svg";

export type Connector = { id: string; label: string; icon: string; mcp: string };

/** Display order follows the composer row: chat, mail, calendars, files, dev. */
export const CONNECTORS: Connector[] = [
  { id: "slack", label: "Slack", icon: slack, mcp: "mcp.slack.com" },
  { id: "gmail", label: "Gmail", icon: gmail, mcp: "gmailmcp.googleapis.com" },
  { id: "calcom", label: "Cal.com", icon: calcom, mcp: "mcp.cal.com" },
  { id: "google-calendar", label: "Google Calendar", icon: googleCalendar, mcp: "calendarmcp.googleapis.com" },
  { id: "google-drive", label: "Google Drive", icon: googleDrive, mcp: "drivemcp.googleapis.com" },
  { id: "microsoft-teams", label: "Microsoft Teams", icon: microsoftTeams, mcp: "mcp.microsoft.com" },
  { id: "dropbox", label: "Dropbox", icon: dropbox, mcp: "mcp.dropbox.com" },
  { id: "stripe", label: "Stripe", icon: stripe, mcp: "mcp.stripe.com" },
  { id: "asana", label: "Asana", icon: asana, mcp: "mcp.asana.com" },
  { id: "box", label: "Box", icon: box, mcp: "mcp.box.com" },
  { id: "github", label: "GitHub", icon: github, mcp: "api.githubcopilot.com" },
  { id: "jira", label: "Jira", icon: jira, mcp: "mcp.atlassian.com" },
  { id: "linear", label: "Linear", icon: linear, mcp: "mcp.linear.app" },
  { id: "salesforce", label: "Salesforce", icon: salesforce, mcp: "api.salesforce.com" },
];

export function ConnectorStrip() {
  const [on, setOn] = createSignal<string[]>([]);
  const [open, setOpen] = createSignal(true);

  function toggle(id: string): void {
    setOn((cur) => (cur.includes(id) ? cur.filter((c) => c !== id) : [...cur, id]));
  }

  return (
    <Show when={open()}>
      <div class="flex items-center gap-2 border-t border-white/[0.06] px-3 py-2.5">
        <p class="min-w-0 truncate text-[11px] text-white/45">Get better answers from your apps</p>
        <div class="ml-auto flex shrink-0 items-center gap-1">
          <For each={CONNECTORS}>
            {(c) => (
              <button
                title={`${c.label} — ${c.mcp} (MCP connector)`}
                aria-label={`${c.label} connector`}
                aria-pressed={on().includes(c.id)}
                onClick={() => toggle(c.id)}
                class={`flex h-6 w-6 items-center justify-center rounded-md p-0.5 transition ${
                  on().includes(c.id)
                    ? "bg-white/15 ring-1 ring-white/25"
                    : "opacity-70 hover:bg-white/10 hover:opacity-100"
                }`}
              >
                <img
                  src={c.icon}
                  alt=""
                  draggable={false}
                  class={`h-full w-full object-contain ${c.id === "github" ? "rounded-full bg-white" : ""}`}
                />
              </button>
            )}
          </For>
          <button
            title="Hide connectors"
            aria-label="Hide connectors"
            onClick={() => setOpen(false)}
            class="ml-0.5 flex h-6 w-6 items-center justify-center rounded-md text-white/40 transition hover:bg-white/10 hover:text-white/80"
          >
            <svg viewBox="0 0 20 20" class="h-3.5 w-3.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
              <path d="M5 5l10 10M15 5L5 15" />
            </svg>
          </button>
        </div>
      </div>
    </Show>
  );
}
