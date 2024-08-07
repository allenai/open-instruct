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

      const handleEsc = (event: KeyboardEvent) => {
        if (event.key === 'Escape') {
          onClose();
        }
      };
      window.addEventListener('keydown', handleEsc);

      // Cleanup event listener on component unmount or when modal closes
      return () => {
        window.removeEventListener('keydown', handleEsc);
      };
    } else {
      setTimeout(() => setShowModal(false), 400); // Match this duration with your transition duration
    }
  }, [isOpen, onClose]);

  if (!showModal) return null;


  const handleOverlayClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
    className={`fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50 transition-opacity duration-400 ${isOpen ? 'opacity-100' : 'opacity-0'}`}
    onClick={handleOverlayClick}
    >
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
