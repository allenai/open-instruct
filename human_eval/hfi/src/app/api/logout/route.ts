// app/api/logout/route.ts
import { NextRequest, NextResponse } from 'next/server';

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

  const response = NextResponse.json({ message: 'Logout successful' });

  cookiesToClear.forEach((cookieName) => {
    response.cookies.set(cookieName, '', {
      maxAge: -1,
      path: '/flask',
    });
  });

  return response;
};