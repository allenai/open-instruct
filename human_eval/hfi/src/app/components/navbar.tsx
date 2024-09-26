"use client";
import React, { FC, useState } from 'react';
import Modal from './modal';
import { useRouter } from 'next/navigation';
import useFlaskService from '../services/flask.service';
import { lastValueFrom } from 'rxjs';

type NavBarProps = {
  openShortcutsModal(): void,
  closeShortcutsModal(): void,
  isShortcutsModalOpen: boolean,
}

const Navbar : FC<NavBarProps> = ({ openShortcutsModal, closeShortcutsModal, isShortcutsModalOpen }) => {

  const [isFeedbackModalOpen, setFeedbackModalOpen] = useState(false);

  const openFeedbackModal = () => setFeedbackModalOpen(true);
  const closeFeedbackModal = () => setFeedbackModalOpen(false);
  const flask = useFlaskService()

  const router = useRouter()

  const logout = async () => {
    try {
        await lastValueFrom(flask.getLogout());
    } catch (error) {
      const errorMessage = (error as any).message as string;
      if (!errorMessage.includes('login 404 (Not Found)')) {
        console.error('Failed to log out', error);
      }
    } finally {
        // Clear cookies after logout
        console.log('logged out!');
        
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
      
        cookiesToClear.forEach(cookieName => {
          document.cookie = `${cookieName}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=None; Secure`;
        });
      
        
        router.push('/')
    }
};

  return (
    <nav className="bg-gray-800 p-4">
      <div className="container mx-auto flex justify-between items-center">
          <a className="text-white text-lg font-bold cursor-pointer" >🕵 &nbsp; Human Evaluation</a>
        <div className="space-x-8 text-sm">
          <a href="#" onClick={openFeedbackModal} className="text-gray-300 hover:text-white">
            Report
          </a>
          <a href="#" onClick={openShortcutsModal} className="text-gray-300 hover:text-white">
            Shortcuts
          </a>
          <a href="#" onClick={logout} className="text-gray-300 hover:text-white">
            Log out
          </a>
        </div>
      </div>
      <Modal isOpen={isFeedbackModalOpen} onClose={closeFeedbackModal}>
        <h2 className="text-xl font-bold mb-4">Instance feedback</h2>
        <p className="mb-4">Do you find the instance interesting, invalid, or too hard to complete? Please let us know by giving feedback here!</p>
        <div className="form-check form-check-inline">
            <input className="form-check-input m-2" type="radio" name="instance-quality" id="instance-quality-good" value="good" />
            <label className="form-check-label" htmlFor="instance-quality-good">This example is interesting.</label>
        </div>
        <div className="form-check form-check-inline">
            <input className="form-check-input m-2" type="radio" name="instance-quality" id="instance-quality-bad" value="bad" />
            <label className="form-check-label" htmlFor="instance-quality-bad">This example is invalid.</label>
        </div>
        <div className="form-check form-check-inline">
            <input className="form-check-input m-2" type="radio" name="instance-quality" id="instance-quality-hard" value="hard" />
            <label className="form-check-label" htmlFor="instance-quality-bad">This example is too hard for me.</label>
        </div>
        <div className="form-group mt-4 flex flex-col">
            <label htmlFor="comment">Comment:</label>
            <textarea className="form-control border-gray border rounded mb-4" id="comment" name="comment" rows={2}></textarea>
        </div>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded mr-4"
        >
          Submit
        </button>
        <button
          className="px-4 py-2 bg-gray-500 text-white rounded"
          onClick={closeFeedbackModal}
        >
          Cancel
        </button>
      </Modal>
      <Modal isOpen={isShortcutsModalOpen} onClose={closeShortcutsModal}>
        <h2 className="text-xl font-bold mb-4">Keyboard Shortcuts</h2>
        <p className="mb-4">To make your annotation experience smoother, use the following keyboard shortcuts.</p>
        <div className="my-4 flex flex-col">
        <ul>
       

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
             👍 - Output A is acceptable. This shortcut <kbd>i</kbd>, along with shortcut <kbd>r</kbd> is at the top level keyboard.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                u
              </span>
            </span>
          </li>
          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
            👎 - Output A is not acceptable. This shortcut <kbd>u</kbd>, along with shortcut <kbd>t</kbd> is at the top level keyboard.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                i
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
             👍 - Output B is acceptable. This shortcut <kbd>k</kbd>, along with shortcut <kbd>f</kbd> is at the bottom level keyboard.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                j
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
            👎 - Output B is not acceptable. This shortcut <kbd>j</kbd>, along with shortcut <kbd>g</kbd> is at the bottom level keyboard.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                k
              </span>
            </span>
          </li>
          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
            A is clearly better. This shortcut maps from 1 to 5, ranging from A to B.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                1
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
            B is clearly better. This shortcut maps from 1 to 5, ranging from A to B.
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                5
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
              Save
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl or Cmd
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                s
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
              Go to previous instance
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl or Cmd
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                j
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
              Go to next instance
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl or Cmd
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                k
              </span>
            </span>
          </li>
          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-gray-800">
              Go to next instance (if saved)
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                space
              </span>
            </span>
          </li>
        </ul>
        </div>
        <button
          className="px-4 py-2 bg-gray-500 text-white rounded"
          onClick={closeShortcutsModal}
        >
          Okay
        </button>
      </Modal>
    </nav>
  );
};

export default Navbar;