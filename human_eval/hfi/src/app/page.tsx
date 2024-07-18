import dynamic from 'next/dynamic'
import { cookies } from 'next/headers';
import { useEffect, useState } from 'react';
const LoginNoSSR = dynamic(() => import('./components/login-page'), { ssr: false })


export default function Home() {
    const backendUrl = process.env.PYTHON_HUMAN_EVAL_BACKEND_URL!


    return (
        <>
        <LoginNoSSR backendUrl={backendUrl} />
        </>
    );
}
