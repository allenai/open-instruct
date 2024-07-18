import { useEffect, useState } from "react";
import { SingleSelectQuestion, SingleSelectQuestionParams } from "../components/question";
import { HandleKeyboardShortcut, KeyboardShortcutParams } from "./evaluation.shortcuts";

type FormState = {
  a_acceptable: string,
  b_acceptable: string,
  rank: string
}

export default function Evaluation() {

  const [currentQuestion, setCurrentQuestion] = useState(1)

  const [formState, setFormState] = useState<FormState>({} as FormState)

  const q3Params: SingleSelectQuestionParams = {
    id: 'rank',
    selectedValue: formState.rank,
    question: "Please select your preference:",
    options: [
      { label: "A is clearly better", value: "1" },
      { label: "A is slightly better", value: "2" },
      { label: "Tie", value: "3" },
      { label: "B is slightly better", value: "4" },
      { label: "B is clearly better", value: "5" }
    ]
  };


  useEffect(() => {

    const set = (q: string, v: string) => {
      setFormState({
        ...formState,
        [q]: v
      })
      console.log(formState)
    }

    const setEvalForQuestion = (v: string) => {
      if (currentQuestion === 1) {
        set('a_acceptable', v)
      } else if (currentQuestion === 2) {
        set('b_acceptable', v)
      }
    }

    const onKeyDown = HandleKeyboardShortcut({
      nextQuestion: () => { setCurrentQuestion(2); console.log('q', currentQuestion) },
      previousQuestion: () => { setCurrentQuestion(1); console.log('q', currentQuestion) },
      nextInstance: () => { console.log('nextInstance') },
      previousInstance: () => { console.log('previousInstance') },
      save: () => { console.log('save') },
      approve: () => { setEvalForQuestion('yes') },
      reject: () => { setEvalForQuestion('no') },
      rank: (r) => { set('rank', String(r)) }
    } as KeyboardShortcutParams);
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
      
  }, [currentQuestion, formState]);

  const Form = () => {
    
    const submit = (e: any) => {
      e.preventDefault()
      setCurrentQuestion(currentQuestion + 1)
      console.log(formState)
    }

    return (
      <>
      <div className={currentQuestion === 1 ? 'border border-red-300 border-2 p-2 rounded my-2' : 'border border-gray-300 border-2 p-2 rounded my-2'}>
        <SingleSelectQuestion 
          id="a_acceptable"
          question="Q1: Is output A an acceptable response?" 
          description="An acceptable response should ① answer the user requests ② have no significant errors ③ have no meaningless text (e.g., repetition)."
          options={ [{label: "Yes", value: "yes"}, {label: "No", value: "no"}]}
          selectedValue={formState.a_acceptable}
          onValueChanged={(v) => setFormState({...formState, a_acceptable: v})}
        />
      </div>
      <div className={currentQuestion === 2 ? 'border border-red-300 border-2 p-2 rounded my-2' : 'border border-gray-300 border-2 p-2 rounded my-2'}>
        <SingleSelectQuestion 
          id="b_acceptable"
          question="Q2: Is output B an acceptable response?" 
          description="An acceptable response should ① answer the user requests ② have no significant errors ③ have no meaningless text (e.g., repetition)."
          options={ [{label: "Yes", value: "yes"}, {label: "No", value: "no"}]}
          selectedValue={formState.b_acceptable}
          onValueChanged={(v) => setFormState({...formState, b_acceptable: v})}
        />
      </div>
      <SingleSelectQuestion 
        id="rank"
        question="Q3: Please choose the response that you prefer (based on helpfulness)." 
        options={q3Params.options}
        selectedValue={formState.rank}
        onValueChanged={(v) => setFormState({...formState, rank: v})}
      />
      <div className="my-4 text-center">
          <input type="submit" onClick={submit} value='Submit' className="btn btn-primary w-fit border-black border rounded px-4 py-2" id="evaluation-submit" />
      </div>
      </>
    )
  }


  return (
    <div id="evaluation-region" className="flex flex-col m-4 p-4 rounded w-full">
      <h2>Evaluation</h2>
      <p className="text-sm text-gray">Now please evaluate the two outputs based on your knowledge, preference, and any external tools (e.g., Google Search or Translate)</p>
      <div className="row mt-3">
          <div className="col icon-col">
          </div>
          <div className="col">
              <form className="text-base" action={() => {}}>
                <Form />
              </form>
          </div>
      </div>
  </div>
  )
}