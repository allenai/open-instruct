import { KeyboardEventHandler, useEffect, useState } from "react";
import { SingleSelectQuestion, SingleSelectQuestionParams } from "../components/question";
import { HandleKeyboardShortcut, KeyboardShortcutParams } from "./evaluation.shortcuts";

export default function Evaluation() {

  const [currentQuestion, setCurrentQuestion] = useState(1)

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


  useEffect(() => {
    const onKeyDown = HandleKeyboardShortcut({
      nextQuestion: () => { setCurrentQuestion(currentQuestion + 1); console.log('q', currentQuestion) },
      previousQuestion: () => { setCurrentQuestion(currentQuestion - 1); console.log('q', currentQuestion) },
      nextInstance: () => { console.log('nextInstance') },
      previousInstance: () => { console.log('previousInstance') },
      save: () => { console.log('save') },
      approve: () => { console.log('approve') },
      reject: () => { console.log('reject') },
      rank: (r) => { console.log('rank', r) }
    } as KeyboardShortcutParams);
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
      
  }, [currentQuestion]);



  return (
    <div id="evaluation-region" className="flex flex-col m-4 p-4 rounded w-full">
      <h2>Evaluation</h2>
      <p className="text-sm text-gray">Now please evaluate the two outputs based on your knowledge, preference, and any external tools (e.g., Google Search or Translate)</p>
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