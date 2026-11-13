import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App.tsx';
import { FlowApp } from './FlowApp.tsx';
import { LiveApp } from './LiveApp.tsx';
import { MockApp } from './MockApp.tsx';
import './styles.css';

// / is the live six-screen flow; ?story the earlier scroll story; ?mock the owner's mockup; ?test the older run page
const q = new URLSearchParams(window.location.search);
createRoot(document.getElementById('root') as HTMLElement).render(
  <StrictMode>{q.has('mock') ? <MockApp /> : q.has('test') ? <FlowApp /> : q.has('story') ? <App /> : <LiveApp />}</StrictMode>,
);
