"use client"
import { useEffect, useState } from "react";

import Form from './login-form';
import { useRouter } from "next/navigation";

export default function Login({backendUrl}: {backendUrl: string}) {

    const [formState, setFormState] = useState<'login'|'signup'>('login')
    const router = useRouter()

    const toggleState = () => {
        const newState = (formState === 'login') ? 'signup' : 'login';
        setFormState(newState);
    }

    const [sessionData, setSessionData] = useState<string | null>(null);

    useEffect(() => {
        const fetchSessionData = async () => {
            try {
                const response = await fetch('/flask/api/user', {
                credentials: 'include',
            });
            const data = await response.json();
            if (response.ok && data.username) {
                router.push('/fine-tuning/1')
                return;
            }
            setSessionData(data.user);
            
            } catch (error) {
            console.error('Failed to fetch session data', error);
            }
        };
    
    fetchSessionData();
    }, [router]);
    
    
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
