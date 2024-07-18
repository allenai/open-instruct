"use client";
import React, { useState } from 'react';
import Modal from './modal';

export default function Navbar() {

  const [isFeedbackModalOpen, setFeedbackModalOpen] = useState(false);

  const openFeedbackModal = () => setFeedbackModalOpen(true);
  const closeFeedbackModal = () => setFeedbackModalOpen(false);

  const [isShortcutsModalOpen, setShortcutsModalOpen] = useState(false);

  const openShortcutsModal = () => setShortcutsModalOpen(true);
  const closeShortcutsModal = () => setShortcutsModalOpen(false);


  return (
    <nav className="bg-gray-800 p-4">
      <div className="container mx-auto flex justify-between items-center">
          <a className="text-white text-lg font-bold" href='/fine-tuning'>🕵 &nbsp; Human Evaluation</a>
        <div className="space-x-8 text-sm">
          <a href="#" onClick={openFeedbackModal} className="text-gray-300 hover:text-white">
            Report
          </a>
          <a href="#" onClick={openShortcutsModal} className="text-gray-300 hover:text-white">
            Shortcuts
          </a>
          <a href="#" className="text-gray-300 hover:text-white">
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
            <span className="font-bold text-gray-800">
              Bold
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                b
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="italic text-gray-800">
              Italic
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                i
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="underline underline-offset-4 text-gray-800">
              Underline
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                u
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="line-through text-gray-800">
              Strikethrough
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Alt
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                u
              </span>
            </span>
          </li>

          <li className="flex justify-between items-center gap-x-2 py-2.5 border-t border-gray-200 first:border-t-0 dark:border-neutral-700">
            <span className="text-sm text-gray-800">
              Small text
            </span>
            <span className="flex flex-wrap items-center gap-x-1 text-sm text-gray-600 dark:text-neutral-400">
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                Ctrl
              </span>
              +
              <span className="min-h-[30px] inline-flex justify-center items-center py-1 px-1.5 bg-gray-200 border border-transparent font-mono text-sm text-gray-800 rounded-md dark:bg-neutral-700 dark:text-neutral-200">
                s
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