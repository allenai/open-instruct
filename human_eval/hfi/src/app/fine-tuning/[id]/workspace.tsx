"use client"
import { useEffect, useState } from "react";
import Evaluation from "./evaluation";
import InstructionAndInput from "./instruction-and-input";
import useFlaskService, { EvaluationInput, ModelOutput as ModelOutputType } from "../../services/flask.service";
import { firstValueFrom, lastValueFrom } from "rxjs";
import ModelOutput from "./model-output";
import { useRouter } from "next/navigation";

export type WorkspaceParams = {
  instanceId: number
  openShortcutsModal: () => void
}

export default function Workspace({ instanceId, openShortcutsModal }: WorkspaceParams) {

  const [currentInstanceId, setCurrentInstanceId ] = useState(instanceId)
  const [modelOutput, setModelOutput] = useState<ModelOutputType|null>(null)
  const [username, setUsername] = useState<string>('')
  const flask = useFlaskService()
  
  const router = useRouter()

  const next = () => {
    setCurrentInstanceId(currentInstanceId + 1)
    router.push(`/fine-tuning/${currentInstanceId + 1}`)
  }

  const prev = () => {
    if (currentInstanceId === 1) return
    setCurrentInstanceId(currentInstanceId - 1)
    router.push(`/fine-tuning/${currentInstanceId - 1}`)
  }

  useEffect(() => {
    const init = async () => {
      if (!instanceId) return;
      if (!flask) return;
      
      try {
        let modelOutput = await firstValueFrom(flask.getModelOutputs(instanceId));
        setModelOutput(modelOutput);
        let authdUser = await firstValueFrom(flask.getAuthenticatedUser())
        setUsername(authdUser)
      } catch (error) {
        if ((error as any).message.includes('Unauthorized')) {
          router.replace('/?unauthorized=true');
        }
      }
    }

    init();
  }, [flask, instanceId]);

  const save = async (input: Partial<EvaluationInput>) => {
    if (!modelOutput) return false;
    if (!input.a_is_acceptable) throw Error('Question 1 is required');
    if (!input.b_is_acceptable) throw Error('Question 2 is required');
    if (!input.rank) throw Error('Question 3 is required');

    // append username
    input = {...input, evaluator: username}

      const save$ = flask.saveModelOutput(instanceId, modelOutput!, {
        a_is_acceptable: input.a_is_acceptable,
        b_is_acceptable: input.b_is_acceptable,
        evaluator: username,
        rank: input.rank
      } as EvaluationInput)

      const response = await lastValueFrom(save$)
      const json = await response.json()
      console.log(json)
      return true;
  }

  return (
    <div className="container flex flex-col lg:flex-row  mx-auto">
      <div className="flex flex-col lg:w-4/6 p-4">
        <InstructionAndInput description="" prompt={modelOutput?.prompt} />
        <ModelOutput modelOutput={modelOutput} />
      </div>
      <div className="flex flex-col lg:w-2/6 p-4">
        <Evaluation username={username} openShortcutsModal={openShortcutsModal} flask={flask} instanceId={currentInstanceId} save={save} nextInstance={next} previousInstance={prev}  />
      </div>
  </div>
  )
}