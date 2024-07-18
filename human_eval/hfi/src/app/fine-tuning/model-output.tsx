import ChatMessage from "../components/chat-message";
import { ModelOutput as ModelOutputType } from "../services/flask.service";

export type ModelOutputParams = {
    modelOutput: ModelOutputType|null
    size?: 'sm'|'xs'|'base'
}

export default function ModelOutput({ modelOutput, size = 'xs' }: ModelOutputParams) {
  return (
    <div id="model-outputs-region" className="flex flex-col m-4 p-4 rounded w-full">
       <ChatMessage icon="🤖" message="Here are some responses from two AI models." />
      <div className="flex flex-col lg:flex-row my-4">
          <div className="flex flex-col lg:w-1/2">
              <div className="d-flex justify-content-center text-center">
                  <button className="completion-icon">A</button>
              </div>
          
              <div className="col completion-col rounded" id="completion-A-col">
                <pre className="whitespace-pre-wrap text-xs">
                    <code>
                    { modelOutput?.completions[0].completion }
                    </code>
                </pre>
              </div>
          </div>
          <div className="flex flex-col lg:w-1/2">
              <div className="d-flex justify-content-center text-center">
                  <button className="completion-icon">B</button>
              </div>
              <div className="col completion-col rounded" id="completion-B-col">
              <pre className="whitespace-pre-wrap text-xs">
                    <code>
                    { modelOutput?.completions[1].completion }
                    </code>
                </pre>
              </div>
          </div>
      </div>
  </div>
  )
}