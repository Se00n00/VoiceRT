import React from "react";
import { render } from "ink";
import { App } from "./app.js";

const seconds = Number(process.argv[2] ?? process.env.VOICE_SECONDS ?? 5) || 5;
render(<App seconds={seconds} />);
