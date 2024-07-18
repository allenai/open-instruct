import { Observable, defer, from , concatMap} from 'rxjs';

export type ModelOutput = {
  completions: {
    completion: string
    model: string
  }[]
  prompt: string
}

export class FlaskService {
  constructor() {

  }

  getModelOutputs(id: number): Observable<ModelOutput>{

    const get$ = defer(() => from(fetch(`/flask/api/model-outputs/${id}`, {
      credentials: 'include',
    })));

    return get$.pipe(concatMap(r => r.json() as Promise<ModelOutput>));
  }
}

const flaskService = new FlaskService();

export default function useFlaskService() {
  return flaskService;
};