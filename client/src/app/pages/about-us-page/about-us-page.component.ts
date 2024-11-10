import { Component } from '@angular/core';

@Component({
  selector: 'app-about-us-page',
  standalone: true,
  imports: [],
  templateUrl: './about-us-page.component.html',
  styles: ``
})
export class AboutUsPageComponent {
  public members: {name: string, github: string, link: string, image: string, pronouns: string, role: string, motto: string}[] = [
    {
      name: 'Șoitu Viorel', github: 'https://github.com/1viorel', 
      link: 'https://1viorel.tech',
      image: 'https://avatars.githubusercontent.com/u/32220246?v=4',
      pronouns: 'drop/table',
      role: 'Web developer',
      motto: 'Can center a div, did the frontend and some of the backend.'
    },
    {
      name: 'Adrian Badea', github: 'https://github.com/AdrianBadea23', 
      link: '',
      image: 'https://i.imgur.com/AekYYua.png',
      pronouns: 'BET/MAN',
      role: 'AI Engineer',
      motto: "I'm ENGI-nearing my fucking limit"
    },
    {
      name: 'Alex Enache', github: '', 
      link: 'https://github.com/tunashred',
      image: 'https://i.imgur.com/EBJVy1e.png',
      pronouns: 'cr/ab',
      role: 'AI engineer',
      motto: 'Regula 1: Orice om poate lucra ca muncitor necalificat in constructii, urmata de Regula 14'
    },
    {
      name: 'Mihai Enache', github: 'https://github.com/tyr4/', 
      link: '',
      image: 'https://i.imgur.com/LKuA38T.jpeg',
      pronouns: 'le/mon',
      role: "Santa's helper",
      motto: 'bUt iT WoRkS oN mY mAcHiNe'
    }
    ]
}
