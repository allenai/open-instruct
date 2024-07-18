import ChatMessage from "../components/chat-message";
import { SingleSelectQuestion, SingleSelectQuestionParams } from "../components/question";

export default function Evaluation() {

const q3Params: SingleSelectQuestionParams = {
  question: "Please select your preference:",
  options: [
    { label: "A is clearly better", value: "a-is-better" },
    { label: "A is slightly better", value: "a-is-slightly-better" },
    { label: "Tie", value: "tie" },
    { label: "B is slightly better", value: "b-is-slightly-better" },
    { label: "B is clearly better", value: "b-is-better" }
  ]
};

  return (
    <div id="evaluation-region" className="flex flex-col m-4 p-4 rounded">
      <ChatMessage icon="📝" message="Now please evaluate the two outputs based on your knowledge, preference, and any external tools (e.g., Google Search or Translate)" />
      <div className="row mt-3">
          <div className="col icon-col">
          </div>
          <div className="col">
              <form className="text-base">
              <SingleSelectQuestion 
                    question="Q1: Is output A an acceptable response?" 
                    description="An acceptable response should ① answer the user requests ② have no significant errors ③ have no meaningless text (e.g., repetition)."
                    options={ [{label: "Yes", value: "yes"}, {label: "No", value: "no"}]}
                  />
                  <SingleSelectQuestion 
                    question="Q2: Is output B an acceptable response?" 
                    description="An acceptable response should ① answer the user requests ② have no significant errors ③ have no meaningless text (e.g., repetition)."
                    options={ [{label: "Yes", value: "yes"}, {label: "No", value: "no"}]}
                  />
                  <SingleSelectQuestion 
                    question="Q3: Please choose the response that you prefer (based on helpfulness)." 
                    options={q3Params.options}
                  />
                  <div className="my-4 text-center">
                      <button type="submit" className="btn btn-primary w-fit border-black border rounded px-4 py-2" id="evaluation-submit">Submit</button>
                  </div>
              </form>
          </div>
      </div>
  </div>
  )
}