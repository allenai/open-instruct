"use client"
import { useEffect, useState } from "react";
import Evaluation from "./evaluation";
import InstructionAndInput from "./instruction-and-input";
import useFlaskService, { ModelOutput as ModelOutputType } from "../../services/flask.service";
import { lastValueFrom } from "rxjs";
import ModelOutput from "./model-output";
import { useRouter } from "next/navigation";

export type WorkspaceParams = {
  instanceId: number
}

export default function Workspace({ instanceId }: WorkspaceParams) {

  const [currentInstanceId, setCurrentInstanceId ] = useState(instanceId)
  const flask = useFlaskService()
  const [modelOutput, setModelOutput] = useState<ModelOutputType|null>(null)
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
    console.log('reloading flask')
    const init = async () => {
      let modelOutput = await lastValueFrom(flask.getModelOutputs(instanceId));
      setModelOutput(modelOutput);
      console.log(modelOutput);
    }

    init();
  }, [flask, instanceId]);

  return (
    <div className="container flex flex-col lg:flex-row  mx-auto">
      <div className="flex flex-col lg:w-4/6 p-4">
        <InstructionAndInput prompt={modelOutput?.prompt} />
        <ModelOutput modelOutput={modelOutput} />
      </div>
      <div className="flex flex-col lg:w-2/6 p-4">
      <Evaluation instanceId={currentInstanceId} nextInstance={next} previousInstance={prev} goToInstance={setCurrentInstanceId} />
      </div>
  </div>
  )
}