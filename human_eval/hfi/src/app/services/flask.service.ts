import { Observable, defer, from , concatMap, map} from 'rxjs';

export type ModelOutput = {
  completions: {
    completion: string
    model: string
  }[]
  prompt: string
}

export type EvaluationInput = {
  a_is_acceptable: string
  b_is_acceptable: string
  evaluator: string
  rank: string
}

export class FlaskService {
  constructor() {

  }

  getAuthenticatedUser(): Observable<string> {
    const get$ = defer(() => from(fetch('/flask/api/user', {
      credentials: 'include',
    })));

    return get$.pipe(
      concatMap(r => r.json() as Promise<{username: string}>),
      map(r => r.username),
    )
  }

  getModelOutputs(id: number): Observable<ModelOutput>{

    const get$ = defer(() => from(fetch(`/flask/api/model-outputs/${id}`, {
      credentials: 'include',
    })));

    return get$.pipe(concatMap(r => r.json() as Promise<ModelOutput>));
  }
  saveModelOutput(id: number, modelOutput: ModelOutput, evaluationInput: EvaluationInput) {
    return from(fetch(`/flask/api/submit-evaluation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        index: id,
        prompt: modelOutput.prompt,
        model_a: modelOutput.completions[0].model,
        model_b: modelOutput.completions[1].model,
        completion_a: modelOutput.completions[0].completion,
        completion_b: modelOutput.completions[1].completion,
        completion_a_is_acceptable: evaluationInput.a_is_acceptable,
        completion_b_is_acceptable: evaluationInput.b_is_acceptable,
        preference: evaluationInput.rank,
        evaluator: evaluationInput.evaluator,

      }),
      credentials: 'include',
    }))
  }
}

const flaskService = new FlaskService();

export default function useFlaskService() {
  return flaskService;
};