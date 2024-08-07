"use client"
import { useEffect, useState } from "react";

import Form from './login-form';
import { useRouter } from "next/navigation";
import useFlaskService from "../services/flask.service";
import { firstValueFrom } from "rxjs";

export type LoginState = 'loading'|'login'|'signup'|'redirecting'

export default function Login({backendUrl}: {backendUrl: string}) {

    const [formState, setFormState] = useState<LoginState>('loading')
    const router = useRouter()
    const flask = useFlaskService()

    const toggleState = () => {
        const newState = (formState === 'login') ? 'signup' : 'login';
        setFormState(newState);
    }

    const [sessionData, setSessionData] = useState<string | null>(null);

    useEffect(() => {
        const getUser = async () => {
            try {
                const user$ = flask.getAuthenticatedUser()
                const user = await firstValueFrom(user$)

                if (user) {
                    setFormState('redirecting')
                    setSessionData(user);
                    router.push('/fine-tuning/1')
                    return;
                } 
            } catch (e) {
                // Python returns a template, not aJSON response
                const knownError = e instanceof Error && e.message.includes('Unexpected token');
                if (!knownError) { console.error(e); }
            }
            setFormState('login')
            console.error('Failed to get authenticated user; please log in');
        };
    
    getUser();
    }, [router, flask]);

    if (formState == 'redirecting') {
        return <>
        <div className="row">
            <div id="nav">
                {/* style="text-decoration: none" */}
                <h2 className="text-center" id="title"><a href="/">🕵 Human Evaluation</a></h2>
                <p className="text-center my-2">Redirecting...</p>
            </div>
        </div>
        </>;
    }
    
    
  return (
    <>
    <div className="row">
        <div id="nav" className="col-md-3">
            {/* style="text-decoration: none" */}
            <h2 className="text-center" id="title"><a href="/">🕵 Human Evaluation</a></h2>
            <div className="row col-12 text-center" id="login-region">
                <Form state={formState} backendUrl={backendUrl} />
                <button onClick={toggleState} className="btn btn-primary m-1">Switch to Signup</button>
                
            </div>
        </div>

        <div className="col-md-9"></div>
    </div>
    </>
  );
}
