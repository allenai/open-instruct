"use client"
export default function Form({ state, backendUrl }: {state: 'login'|'signup', backendUrl: string}) {

  let buttonLabel = (state === "login") ? 'Login' : 'Sign up';
  let formAction = (state === "login") ? '/flask/login' : '/flask/signup';

  return (
  <form className="m-5" id="authForm" action={formAction} method="post">
      <div className="form-group mt-4">
          <label className="mx-4" htmlFor="username">Username</label>
          <input type="text" className="form-control p-2 rounded" id="username" name="username" required />
      </div>
      <div className="form-group mt-4 mb-5">
          <label className="mx-4" htmlFor="password">Password</label>
          <input type="password" className="form-control p-2 rounded" id="password" name="password" required />
      </div>
      <input type="submit" className="btn btn-primary m-1" value={buttonLabel} id="submitBtn" />
  </form>
  );
}