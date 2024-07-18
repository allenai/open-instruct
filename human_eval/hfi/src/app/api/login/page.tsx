// pages/api/session.ts

import axios from 'axios';
import { NextRequest, NextResponse } from 'next/server';

export const GET = async (req: NextRequest) => {
  try {
    console.log('cookies', req.headers.get('cookie'))
    const response = await axios.get(`${process.env.PYTHON_HUMAN_EVAL_BACKEND_URL}/api/user`, {
      headers: {
        'Content-Type': 'application/json',
        Cookie: req.headers.get('cookie') || '', // Forward cookies to Flask backend
      },
      withCredentials: true, // Include credentials (cookies) in the request
    });
    console.log('successfully logged in');
    return NextResponse.redirect('/fine-tuning');
  } catch (error) {
    console.error(error)
    return NextResponse.json({ message: 'Failed to fetch session data' }, {
      status: 500,
    });
  }
};  