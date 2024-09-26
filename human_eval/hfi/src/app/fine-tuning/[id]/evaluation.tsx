import { useEffect, useRef, useState } from "react";
import { SingleSelectQuestion, SingleSelectQuestionParams } from "../../components/question";
import { HandleKeyboardShortcut, KeyboardShortcutParams } from "./evaluation.shortcuts";
import { EvaluationInput, FlaskService, ModelOutput } from "@/app/services/flask.service";

type FormState = {
  a_acceptable: string,
  b_acceptable: string,
  rank: string
}

type EvaluationParams = {
  instanceId: number
  username: string
  nextInstance: () => void
  previousInstance: () => void
  openShortcutsModal: () => void
  save: (input: Partial<EvaluationInput>) => Promise<boolean>
  flask: FlaskService
}

export default function Evaluation({ username, instanceId, save, nextInstance, previousInstance, flask, openShortcutsModal } : EvaluationParams) {

  const [currentQuestion, setCurrentQuestion] = useState(1)
  const [isSaved, setIsSaved] = useState(false)
  const [formState, setFormState] = useState<FormState>({} as FormState)
  const submitRef = useRef<(any)>(null);

  const q3Params: SingleSelectQuestionParams = {
    id: 'rank',
    selectedValue: formState.rank,
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
    const set = (q: string, v: string) => {
      setFormState({
        ...formState,
        [q]: v
      })
    }

    const setEvalForQuestion = (currentQuestion: number, v: string) => {
      if (currentQuestion === 1) {
        set('a_acceptable', v)
      } else if (currentQuestion === 2) {
        set('b_acceptable', v)
      }
    }

    const beginSave = async () => {
      console.log('beginning save')
      const result = await save({
        a_is_acceptable: formState.a_acceptable,
        b_is_acceptable: formState.b_acceptable,
        rank: formState.rank,
      } as Partial<EvaluationInput>)
      
      if (result)
        setIsSaved(true)
    }

    submitRef.current = beginSave

    const shakeSubmit = () => {
      
      const s = document.getElementById("save");
      if (s?.classList.contains("border-red-500") || s?.classList.contains("border-blue-500")) {
        return
      }
      s?.classList.remove("border-black");
      s?.classList.add("border-red-500", "text-red-500");

      setTimeout(() => {
        s?.classList.add("border-black");
        s?.classList.remove("border-red-500", "text-red-500");
      }, 800);
    }

    const onKeyDown = HandleKeyboardShortcut({
      previousQuestion: () => { setCurrentQuestion(1); },
      nextQuestion: () => { setCurrentQuestion(2); },
      nextInstance: () => { nextInstance() },
      previousInstance: () => { previousInstance() },
      nextInstanceIfSaved: () => { if (isSaved) { nextInstance(); } else { shakeSubmit(); } },
      save: beginSave,
      approve: (q) => { setEvalForQuestion(q, 'yes') },
      reject: (q) => { setEvalForQuestion(q, 'no') },
      openShortcuts: openShortcutsModal,
      rank: (r) => { set('rank', r) }
    } as KeyboardShortcutParams);
    window.addEventListener('keydown', onKeyDown);

    return () => {
      window.removeEventListener('keydown', onKeyDown);
    };
      
  }, [currentQuestion, formState, nextInstance, previousInstance, setIsSaved, isSaved, save, openShortcutsModal]);

  const Form = () => {

    const submit = () => {
      if (submitRef && submitRef.current) {
        submitRef.current()
      } 
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
      <div className="my-4 mx-auto text-center flex flex-row">
        <button onClick={previousInstance} className="btn btn-primary w-fit border-gray-400 border text-gray-400 border-2 rounded px-4 py-2 ml-auto mr-2" id="evaluation-submit">Previous</button>
        <button onClick={nextInstance} hidden={isSaved} className="btn btn-primary w-fit border-gray-400 border text-gray-400 border-2 rounded px-4 py-2 mx-2" id="evaluation-submit" >Next</button>
        <input id="save" type="submit" onClick={submit} value={!isSaved ? 'Submit' : 'Next'} className="btn btn-primary w-fit border-black border border-2 cursor-pointer rounded px-4 py-2 mr-1 ml-2" />
      </div>
      </>
    )
  }


  return (
    <div id="evaluation-region" className="flex flex-col m-4 p-4 rounded w-full">
      <h2>Evaluation</h2>
      <p className="text-sm text-gray">Welcome, {username}! Please evaluate the two outputs based on your knowledge, preference, and any external tools (e.g., Google Search or Translate)</p>
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