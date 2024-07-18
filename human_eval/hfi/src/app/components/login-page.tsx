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
                router.push('/fine-tuning')
                return;
            }
            setSessionData(data.user);
            console.log('session data', data);
            } catch (error) {
            console.error('Failed to fetch session data', error);
            }
        };
    
    fetchSessionData();
    }, []);

    
    useEffect(() => {
        const fetchUserData = async () => {
            try {
                const response = await fetch('/flask/api/user', {
                credentials: 'include',
            });
            const data = await response.json();
            console.log('user', data);
            } catch (error) {
            console.error('Failed to fetch user', error);
            }
        };
    
    fetchUserData();
    }, []);
    
    const logout = async () => {
        try {
            const response = await fetch('/api/logout', {
                credentials: 'include',
                method: 'POST',
            });
            await response.text();
            console.log('logged out', response.ok);
        } catch (error) {
            console.error('Failed to log out', error);
        }
    };


    
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
            <div className="row col-12 mt-4 text-center" id="login-region">
                <button onClick={logout} className="btn btn-primary m-1">Log out</button>
            </div>
        </div>

        <div className="col-md-9"></div>
    </div>
    </>
  );
}
