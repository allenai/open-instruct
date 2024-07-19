import { useRouter } from "next/router";
import Navbar from "../../components/navbar";
import Workspace from "./workspace";
import { FC } from 'react';

type HomeParams = {
  params: {
    id: string
  }
}

const Home : FC<HomeParams> = ({ params }) => {
  
  return (
    <>
    <Navbar />
    <div className="flex flex-row w-full">
        <Workspace username="darrensapalo" instanceId={Number(params.id)} />
    </div>
    </>
  );
}

export default Home;