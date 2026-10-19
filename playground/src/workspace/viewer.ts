import { json } from "@codemirror/lang-json";
import { EditorState } from "@codemirror/state";
import { EditorView, lineNumbers } from "@codemirror/view";
import { codeHighlighting } from "./highlight.ts";

/** Read-only, focusable JSON view. */
export const createJsonViewer = (parent: HTMLElement, label: string) => {
  const view = new EditorView({
    parent,
    state: EditorState.create({
      doc: "",
      extensions: [
        EditorState.readOnly.of(true),
        EditorView.editable.of(false),
        lineNumbers(),
        json(),
        codeHighlighting,
        EditorView.contentAttributes.of({ "aria-label": label, tabindex: "0" }),
      ],
    }),
  });
  return {
    show(text: string) {
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
      view.scrollDOM.scrollTop = 0;
    },
    text: () => view.state.doc.toString(),
  };
};
