"use client"
import Navbar from "../../components/navbar";
import Workspace from "./workspace";
import { FC, useState } from 'react';

type HomeParams = {
  params: {
    id: string
  }
}

const Home : FC<HomeParams> = ({ params }) => {

  const [isShortcutsModalOpen, setShortcutsModalOpen] = useState(false);

  const openShortcutsModal = () => setShortcutsModalOpen(true);
  const closeShortcutsModal = () => setShortcutsModalOpen(false);


  return (
    <>
    <Navbar closeShortcutsModal={closeShortcutsModal} openShortcutsModal={openShortcutsModal} isShortcutsModalOpen={isShortcutsModalOpen} />
    <div className="flex flex-row w-full">
        <Workspace openShortcutsModal={openShortcutsModal} instanceId={Number(params.id)} />
    </div>
    </>
  );
}

export default Home;