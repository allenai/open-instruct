// src/components/Modal.tsx

import React, { useEffect, useState } from 'react';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  children: React.ReactNode;
}

const Modal: React.FC<ModalProps> = ({ isOpen, onClose, children }) => {
  const [showModal, setShowModal] = useState(isOpen);

  useEffect(() => {
    if (isOpen) {
      setShowModal(true);
    } else {
      setTimeout(() => setShowModal(false), 400); // Match this duration with your transition duration
    }
  }, [isOpen]);

  if (!showModal) return null;

  return (
    <div className={`fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 transition-opacity duration-400 ${isOpen ? 'opacity-100' : 'opacity-0'}`}>
      <div className="bg-white rounded-lg shadow-lg overflow-hidden w-11/12 sm:w-5/6 md:w-2/3">
        <div className="flex justify-end p-2">
          <button
            className="text-gray-500 hover:text-gray-700 text-lg h-8 w-8"
            onClick={onClose}
          >
            &times;
          </button>
        </div>
        <div className="h-fit px-4 pb-4">
          {children}
        </div>
      </div>
    </div>
  );
};

export default Modal;
