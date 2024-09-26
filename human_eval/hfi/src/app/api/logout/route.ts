// app/api/logout/route.ts
import { NextRequest, NextResponse } from 'next/server';
import axios from 'axios';

export const POST = async (req: NextRequest) => {
  const cookiesToClear = [
    'auth_verification',
    'twk_idm_key',
    'TawkConnectionTime',
    'JSESSIONID',
    'next-auth.csrf-token',
    'next-auth.callback-url',
    'next-auth.session-token',
    'session',
  ];

  const logoutResponse = await logout(req)
  const response = NextResponse.json({ message: 'Logout successful' });

  cookiesToClear.forEach((cookieName) => {
    response.cookies.set(cookieName, '', {
      maxAge: -1,
      path: '/flask',
    });
  });

  return response;
};

async function logout(req: NextRequest) {
  try {
    const response = await axios.get(`${process.env.PYTHON_HUMAN_EVAL_BACKEND_URL}/logout`, {
      headers: {
        'Content-Type': 'application/json',
        Cookie: req.headers.get('cookie') || '', // Forward cookies to Flask backend
      },
      withCredentials: true, // Include credentials (cookies) in the request
    });
    return NextResponse.json({ message: 'OK' });
  } catch (error) {
    console.error(error)
    return NextResponse.json({ message: 'Failed to log out' }, {
      status: 500,
    });
  }
}