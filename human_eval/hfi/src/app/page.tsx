import dynamic from 'next/dynamic'
const LoginNoSSR = dynamic(() => import('./login/login-page'), { ssr: false })

export default function Home() {
    const backendUrl = process.env.PYTHON_HUMAN_EVAL_BACKEND_URL!

    return (
        <>
        <LoginNoSSR backendUrl={backendUrl} />
        </>
    );
}
