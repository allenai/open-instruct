"use client"
import { useEffect, useState } from "react";
import Evaluation from "./evaluation";
import InstructionAndInput from "./instruction-and-input";
import useFlaskService, { ModelOutput as ModelOutputType } from "../services/flask.service";
import { lastValueFrom } from "rxjs";
import ModelOutput from "./model-output";

export type WorkspaceParams = {
  id: number
}

export default function Workspace({ id }: WorkspaceParams) {

  const flask = useFlaskService()

  const [modelOutput, setModelOutput] = useState<ModelOutputType|null>(null)

  useEffect(() => {
    const init = async () => {
      let modelOutput = await lastValueFrom(flask.getModelOutputs(id));
      setModelOutput(modelOutput);
      console.log(modelOutput);
    }

    init();
  }, [flask, id]);

  return (
    <div className="container flex flex-col lg:flex-row  mx-auto">
      <div className="flex flex-col lg:w-4/6 p-4">
        <InstructionAndInput prompt={modelOutput?.prompt} />
        <ModelOutput modelOutput={modelOutput} />
      </div>
      <div className="flex flex-col lg:w-2/6 p-4">
      <Evaluation />
      </div>
  </div>
  )
}